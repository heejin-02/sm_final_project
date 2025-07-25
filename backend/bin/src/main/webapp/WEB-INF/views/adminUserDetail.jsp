<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
</head>
<body>
	<h2>고객 농장 상세 정보</h2>
	<p>아이디 : ${user.userPhone } 이름 : ${user.userName } 전화번호 : ${user.userPhone} </p>
	
	<table border="1">
	    <tr>
	    	<th>번호</th>	
	    	<th>농장 구분</th>
	    	<th>농장 이름</th>
	    	<th>농장 주소</th>
	    	<th>농장 번호</th>
	    	<th>가입날짜</th>
	    	<th>수정/삭제</th>
	    </tr>
	    <tr>
	    	<td>1</td>
	    	<td>${user.farmCrops} </td>
    		<td>${user.farmName}</td>
	    	<td>${user.farmAddr}</td>
	    	<td>${user.farmPhone}</td>
	    	<td>${user.joinedAt}</td>
			<td><button>수정</button><button>삭제</button></td>
	    </tr>
	</table>
	
	
	<a href="/admin">목록으로 돌아가기</a>
	<button>농장 추가</button>
	

</body>
</html>